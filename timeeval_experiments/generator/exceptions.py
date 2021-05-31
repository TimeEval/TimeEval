class AlgorithmManifestLoadingWarning(RuntimeWarning):
    def __init__(self, msg):
        super().__init__(msg)


class MissingManifestWarning(AlgorithmManifestLoadingWarning):
    @staticmethod
    def msg(name: str):
        return f"Algorithm {name} has no manifest! Skipping {name}."


class MissingReadmeWarning(AlgorithmManifestLoadingWarning):
    @staticmethod
    def msg(name: str):
        return f"Algorithm {name} has no README! Skipping {name}."


class InvalidManifestWarning(AlgorithmManifestLoadingWarning):
    @staticmethod
    def msg(name: str, detail_msg: str = ""):
        return f"Algorithm {name}'s manifest is invalid! {detail_msg} Skipping {name}."
