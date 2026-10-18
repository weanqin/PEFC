"""
Unified training infrastructure for MODMA EEG experiments.

Layered architecture:
- BaseTrainer  : epoch loop, History, early stopping, checkpointing
- TrainStep    : per-batch forward + loss (model-specific)
- Callback     : hooks for logging, visualization, etc.
"""

import json
import os
import torch


# =============================================================================
# Callbacks
# =============================================================================

class Callback:
    """Base callback. Subclass and override the hooks you need."""

    def on_train_begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        pass

    def on_train_end(self, trainer, history):
        pass


class PrintingCallback(Callback):
    """Prints metrics at the end of each epoch.

    ``prefix`` is prepended to the line (e.g. ``"[stage1] "`` for multi-stage
    training where two trainers run in sequence). The val line keys off the
    TRAIN metric keys (so a val dict missing some train keys — e.g. a stage-2
    loss that is only computed on train — prints ``0.000`` rather than
    KeyError).
    """

    def __init__(self, print_every=1, prefix=""):
        self.print_every = print_every
        self.prefix = prefix

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        if (epoch + 1) % self.print_every != 0:
            return
        keys = list(train_metrics.keys())
        train_str = " ".join(f"{k}={train_metrics[k]:.3f}" for k in keys)
        val_str = " ".join(f"{k}={val_metrics.get(k, 0):.3f}" for k in keys)
        label = f"{self.prefix}E{epoch + 1}" if self.prefix else f"Epoch {epoch + 1}/{trainer.n_epochs}"
        print(f"{label}: train | {train_str} || val | {val_str}")


# =============================================================================
# History
# =============================================================================

class History:
    """Aggregates per-epoch metrics and supports serialization."""

    def __init__(self):
        self.train = []  # list[dict] — one entry per epoch
        self.val = []    # list[dict]

    @property
    def best_epoch(self):
        """Index (0-based) of the epoch with lowest val loss."""
        if not self.val:
            return None
        return min(
            range(len(self.val)),
            key=lambda i: self.val[i].get("loss", float("inf")),
        )

    def get_curve(self, key):
        """Return a list of values for *key* from the val history."""
        return [epoch.get(key) for epoch in self.val]

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"train": self.train, "val": self.val}, f, indent=2)

    @classmethod
    def from_json(cls, path):
        hist = cls()
        with open(path, "r") as f:
            data = json.load(f)
        hist.train = data.get("train", [])
        hist.val = data.get("val", [])
        return hist


# =============================================================================
# TrainStep — model-specific forward + loss
# =============================================================================

class TrainStep:
    """Per-batch forward + loss computation.

    Subclass this to adapt different model architectures.
    The ``__call__`` receives the raw batch dict and returns
    ``(loss, losses_dict)`` where ``loss`` is a scalar Tensor ready for
    ``.backward()`` and ``losses_dict`` contains per-component Tensors.
    """

    def __call__(self, batch, model, criterion, device):
        raise NotImplementedError


# =============================================================================
# BaseTrainer — model-agnostic training loop
# =============================================================================

class BaseTrainer:
    """Model-agnostic trainer.  The model-specific logic lives in *train_step*.

    Parameters
    ----------
    model : nn.Module
        The model to train.  Interpretation is up to *train_step*.
    criterion : nn.Module or None
        Loss module.  Can be ``None`` if *train_step* handles loss internally.
    device : torch.device
    train_step : TrainStep
        Callable ``(batch, model, criterion, device) -> (loss, losses_dict)``.
    lr : float, default 1e-3
        Adam learning rate (ignored if *optimizer* is supplied).
    optimizer : torch.optim.Optimizer, optional
        External optimizer; overrides the default Adam.
    kl_anneal_epochs : int, default 0
        Number of epochs over which to linearly anneal ``criterion.beta_kl``.
        0 disables annealing.  Only active when *criterion* has ``beta_kl``.
    patience : int, default 20
        Early-stopping patience.  0 disables early stopping.
    save_path : str, default "best_model.pt"
        Where to save the best model state dict.
    max_grad_norm : float, optional
        If given, clip gradient norms.
    use_amp : bool, default False
    amp_dtype : torch.dtype, default bfloat16
    callbacks : list[Callback], optional
    """

    def __init__(
        self,
        model,
        criterion,
        device,
        train_step,
        lr=1e-3,
        optimizer=None,
        kl_anneal_epochs=0,
        patience=20,
        save_path="best_model.pt",
        max_grad_norm=None,
        use_amp=False,
        amp_dtype=torch.bfloat16,
        callbacks=None,
        scheduler=None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.train_step = train_step
        self.kl_anneal_epochs = kl_anneal_epochs
        self.patience = patience
        self.save_path = save_path
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.callbacks = callbacks or []
        self.n_epochs = 0
        self.scheduler = scheduler  # optional LR scheduler; .step() per epoch

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Collect parameters from whatever the step needs
            if hasattr(train_step, "encoder") and hasattr(train_step, "decoder"):
                # AEDiagnosticStep holds encoder/decoder separately
                params = list(train_step.encoder.parameters()) + list(train_step.decoder.parameters())
                self.optimizer = torch.optim.Adam(params, lr=lr)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_callbacks(self, method, *args):
        for cb in self.callbacks:
            getattr(cb, method)(*args)

    @staticmethod
    def _accumulate(totals, key, value):
        totals[key] = totals.get(key, 0.0) + value

    # ------------------------------------------------------------------
    # Public: per-epoch methods
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        """Run one training epoch and return averaged metrics."""
        self.model.train() if hasattr(self.model, "train") else None
        # Handle AEDiagnosticStep where model is None
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train()
        elif hasattr(self.train_step, "encoder"):
            self.train_step.encoder.train()
            self.train_step.decoder.train()

        totals = {}
        n_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                loss, losses_dict = self.train_step(
                    batch, self.model, self.criterion, self.device
                )

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters() if self.model is not None
                    else list(self.train_step.encoder.parameters()) + list(self.train_step.decoder.parameters()),
                    self.max_grad_norm,
                )

            self.optimizer.step()

            self._accumulate(totals, "loss", loss.item())
            for k, v in losses_dict.items():
                self._accumulate(totals, k, v.item())
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    def evaluate(self, dataloader):
        """Run one evaluation pass and return averaged metrics."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        elif hasattr(self.train_step, "encoder"):
            self.train_step.encoder.eval()
            self.train_step.decoder.eval()

        totals = {}
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    loss, losses_dict = self.train_step(
                        batch, self.model, self.criterion, self.device
                    )

                self._accumulate(totals, "loss", loss.item())
                for k, v in losses_dict.items():
                    self._accumulate(totals, k, v.item())
                n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Public: full fit loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader, n_epochs):
        """Train for up to *n_epochs*, with optional early stopping.

        Returns
        -------
        History
            Contains per-epoch train and val metrics.
        """
        self.n_epochs = n_epochs
        history = History()
        best_val_loss = float("inf")
        patience_counter = 0
        original_beta = getattr(self.criterion, "beta_kl", None) if self.criterion is not None else None

        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self._call_callbacks("on_train_begin", self)

        for epoch in range(n_epochs):
            self._call_callbacks("on_epoch_begin", self, epoch)

            # KL annealing
            if (
                self.kl_anneal_epochs > 0
                and original_beta is not None
                and self.criterion is not None
                and hasattr(self.criterion, "beta_kl")
            ):
                if epoch < self.kl_anneal_epochs:
                    self.criterion.beta_kl = original_beta * (
                        epoch / self.kl_anneal_epochs
                    )
                else:
                    self.criterion.beta_kl = original_beta

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history.train.append(train_metrics)
            history.val.append(val_metrics)

            self._call_callbacks(
                "on_epoch_end", self, epoch, train_metrics, val_metrics
            )

            # Early stopping
            if self.patience > 0:
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    # Save state dict. Models that own a checkpoint seam
                    # (e.g. OCLatentFlow.save_checkpoint) write a
                    # self-describing payload; others fall back to a bare
                    # state-dict.
                    if self.model is not None:
                        if hasattr(self.model, "save_checkpoint"):
                            self.model.save_checkpoint(self.save_path)
                        else:
                            torch.save(self.model.state_dict(), self.save_path)
                    elif hasattr(self.train_step, "encoder"):
                        state = {
                            "encoder": self.train_step.encoder.state_dict(),
                            "decoder": self.train_step.decoder.state_dict(),
                        }
                        torch.save(state, self.save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Step the LR scheduler once per epoch (after val). Placed after
            # the early-stop break so a scheduler with T_max=n_epochs still
            # converges on a full run; on early stop the residual schedule is
            # irrelevant (training ends).
            if self.scheduler is not None:
                self.scheduler.step()

        # Restore best model
        if self.patience > 0 and best_val_loss < float("inf"):
            if self.model is not None:
                payload = torch.load(
                    self.save_path,
                    map_location=self.device,
                    weights_only=False,
                )
                # Self-describing payloads (any format_version, e.g. the
                # DysIB v1/v2 checkpoint seams) wrap the state-dict; legacy
                # saves store the state-dict directly. The in-memory model
                # already exists -- only state-dict restoration is needed,
                # not full load_checkpoint.
                state_dict = (
                    payload["state_dict"]
                    if isinstance(payload, dict) and "state_dict" in payload
                    else payload
                )
                self.model.load_state_dict(state_dict)
            elif hasattr(self.train_step, "encoder"):
                state = torch.load(self.save_path, map_location=self.device, weights_only=False)
                self.train_step.encoder.load_state_dict(state["encoder"])
                self.train_step.decoder.load_state_dict(state["decoder"])

        # Restore original beta_kl
        if original_beta is not None and self.criterion is not None and hasattr(self.criterion, "beta_kl"):
            self.criterion.beta_kl = original_beta

        self._call_callbacks("on_train_end", self, history)

        return history


# Backward-compatible alias
Trainer = BaseTrainer
