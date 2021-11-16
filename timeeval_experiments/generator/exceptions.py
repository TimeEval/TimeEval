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
    def msg(name: str, detail_msg: str = "", will_skip: bool = True):
        text = f"Algorithm {name}'s manifest is invalid! {detail_msg}"
        if will_skip:
            text += f"Skipping {name}."
        return text
