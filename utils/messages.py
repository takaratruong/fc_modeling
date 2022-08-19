

def header_message(args):
    print("------------------------------------------------------------------------------")
    print("Exp Name: ", args.exp_name)
    print("Config File:", args.config)
    print("Algorithm:", args.alg)
    print("Environment:", args.environment)
    print("------------------------------------------------------------------------------")

def no_policy():
    print('---------------------------------------------------------------------------------------')
    print('Specify policy path via argument command flag: --policy_path relative/path/to/policy.pt')
    print('Running zero-action policy')
    print('---------------------------------------------------------------------------------------')

