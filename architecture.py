IBM_VIGO = "ibm_vigo"

backends = [IBM_VIGO]


class Architecture:
    def __init__(self, name):
        self.name = name
        if name in backends:
            creater_create_architecture(name)
        else:
            raise KeyError("tmp")


def creater_ibm_vigo_architecture(**kwargs):
    pass


def creater_create_architecture(name, **kwargs):
    if name == IBM_VIGO:
        return creater_ibm_vigo_architecture(**kwargs)
    else:
        raise KeyError("name" + str(name) + "not recognized as architecture name. Please use one of", *backends)
