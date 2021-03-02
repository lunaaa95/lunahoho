import pickle


def savepkl(obj, fname):
    with open(fname, 'wb') as fh:
        pickle.dump(obj, fh)


def loadpkl(fname):
    with open(fname, 'rb') as fh:
        obj = pickle.load(fh)
    return obj
