class BaseSchema:
    """Parent schema for future validation."""
    required_fields = []

    def validate(self, data):
        missing = [f for f in self.required_fields if f not in data]
        return {"valid": len(missing) == 0, "missing": missing}
