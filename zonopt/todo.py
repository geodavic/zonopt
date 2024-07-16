class GeorgePleaseImplement(NotImplementedError):
    def __init__(self, feature: str):
        message = f"The feature `{feature}` is not yet implemented, but will be soon. Please let the repository owner (@geodavic) know that you encountered this message and he will priortize adding it"
        super().__init__(message)
