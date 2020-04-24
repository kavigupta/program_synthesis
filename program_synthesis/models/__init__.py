from .karel_model import KarelLGRLModel, KarelLGRLRefineModel, KarelLGRLOverfitModel
#from .karel_agent import main as rl_main
def get_model(args):
    MODEL_TYPES = {
        'karel-lgrl': KarelLGRLModel,
        'karel-lgrl-ref': KarelLGRLRefineModel,
        'karel-lgrl-overfit' : KarelLGRLOverfitModel
    }
    return MODEL_TYPES[args.model_type](args)
