from .model import Model


def __getattr__(name):
    if name == "starforge":
        from .starforge.starforge import make_model
        model = make_model()
        globals()["starforge"] = model  # cache so it's only built once
        return model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
