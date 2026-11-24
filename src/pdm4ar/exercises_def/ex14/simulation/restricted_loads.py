import io
import pickle


class RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, *, allowed_modules=None, allowed_qualnames=None):
        super().__init__(file)
        self.allowed_modules = set(allowed_modules or [])
        self.allowed_qualnames = set(allowed_qualnames or [])

    def find_class(self, module, name):
        qual = f"{module}.{name}"

        # Allow only if explicitly whitelisted (module OR fully qualified name)
        if (self.allowed_modules and any(module.startswith(m) for m in self.allowed_modules)) or (
            self.allowed_qualnames and qual in self.allowed_qualnames
        ):
            return super().find_class(module, name)

        # Otherwise refuse
        raise pickle.UnpicklingError(f"Disallowed global during load: {qual}")


def restricted_loads(data: bytes, **kwargs):
    """Load with RestrictedUnpickler. kwargs are forwarded to the unpickler constructor."""
    return RestrictedUnpickler(io.BytesIO(data), **kwargs).load()
