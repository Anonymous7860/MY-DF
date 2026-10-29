"""
Optimizer Factory - Configurable Optimizer Selection

Provides a unified interface for creating optimizers with consistent feature parity.
All optimizers support: LR dropout, warmup, cosine annealing, clipnorm.

Usage:
    from core.leras.optimizers import OptimizerFactory

    optimizer = OptimizerFactory.create(
        params=model.parameters(),
        lr=5e-5,
        weight_decay=5e-5,
        ...
    )
"""

from core.leras.optimizers.OptimizerBase import OptimizerBase


class OptimizerRegistry:
    """
    Registry for optimizer classes with decorator-based registration.

    Usage:
        @OptimizerRegistry.register("custom")
        class CustomOptimizer(OptimizerBase):
            ...
    """

    _optimizers: dict[str, type[OptimizerBase]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an optimizer class."""

        def decorator(optimizer_class: type[OptimizerBase]) -> type[OptimizerBase]:
            cls._optimizers[name.lower()] = optimizer_class
            return optimizer_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[OptimizerBase] | None:
        """Get optimizer class by name (case-insensitive)."""
        return cls._optimizers.get(name.lower())

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered optimizer names."""
        return list(cls._optimizers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if optimizer is registered."""
        return name.lower() in cls._optimizers


class OptimizerFactory:
    """
    Factory for creating optimizers with unified configuration.

    Supports: AdaBelief (default), ACProp (async AdaBelief with RAdam rectification)

    All optimizers are created with consistent interface and feature parity:
    - Learning rate dropout (per-step random masks)
    - Linear warmup schedule
    - Cosine annealing schedule
    - Gradient norm clipping
    - Decoupled weight decay (AdamW-style, via weight_decouple=True)
    """

    @classmethod
    def create(
        cls,
        params,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        lr_dropout: float = 1.0,
        lr_cos: int = 0,
        clipnorm: float = 0.0,
        bias_correction: bool = False,
        warmup_steps: int = 0,
        optimizer_type: str = "adabelief",
        name: str | None = None,
        **kwargs,
    ) -> OptimizerBase:
        """
        Create an optimizer instance.

        Args:
            params: Model parameters or parameter groups
            lr: Learning rate
            weight_decay: Weight decay coefficient
            weight_decouple: Decouple weight decay from gradient (AdamW-style, default True)
            lr_dropout: Probability of keeping updates (1.0 = no dropout)
            lr_cos: Cosine annealing period (0 = disabled)
            clipnorm: Maximum gradient norm (0.0 = no clipping)
            bias_correction: Enable Adam bias correction for stable early training
            warmup_steps: Linear warmup steps (0 = disabled)
            optimizer_type: Which optimizer to use ("adabelief" or "acprop")
            name: Optional optimizer name for checkpointing
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Configured optimizer instance
        """
        optimizer_class = OptimizerRegistry.get(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                f"Optimizer '{optimizer_type}' is not registered. Available: {OptimizerRegistry.list_available()}"
            )

        common_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_dropout": lr_dropout,
            "lr_cos": lr_cos,
            "clipnorm": clipnorm,
            "name": name or optimizer_type.capitalize(),
            "bias_correction": bias_correction,
            "warmup_steps": warmup_steps,
            "weight_decouple": weight_decouple,
            "betas": kwargs.get("betas", (0.9, 0.999)),
        }

        if "eps" in kwargs:
            common_kwargs["eps"] = kwargs["eps"]

        # ACProp-specific: rectify defaults to True
        if optimizer_type.lower() == "acprop":
            common_kwargs["rectify"] = kwargs.get("rectify", True)

        return optimizer_class(params, **common_kwargs)

