import argparse
import margin_loss

def get_args(dataset, seed, model_type, train=False, test_only=False, bias=False):
    """Helper function to create an arguments object."""
    return argparse.Namespace(
        dataset=dataset,
        seed=seed,
        type=model_type,
        train=train,
        test_only=test_only,
        bias=bias,
        clustering=False,
        val_only=False,
        gpu='0'
    )

def run_experiments():
    """Main function to run the sequence of experiments."""
    # 반복할 시드 범위
    #seeds = [1834, 3721, 2829, 3049, 5731, 5729, 2194, 4910, 5810, 942]
    #seeds = [4821, 9372, 161, 7059, 2880, 6573, 894, 2134, 7596, 3741]
    seeds = [2411, 5193, 4594]
    dataset = "waterbirds"
    #dataset = "celeba"

    for seed in seeds:
        print(f"\n===== [Seed {seed}] Training Baseline =====")
        baseline_train_args = get_args(dataset, seed, 'baseline', train=True, bias=True)
        margin_loss.main(baseline_train_args)

        print(f"\n===== [Seed {seed}] Training Margin =====")
        margin_train_args = get_args(dataset, seed, 'margin', train=True)
        margin_loss.main(margin_train_args)

        #print(f"\n===== [Seed {seed}] Testing Baseline =====")
        #baseline_test_args = get_args(dataset, seed, 'baseline', test_only=True)
        #margin_loss.main(baseline_test_args)

        print(f"\n===== [Seed {seed}] Testing Margin =====")
        margin_test_args = get_args(dataset, seed, 'margin', test_only=True)
        margin_loss.main(margin_test_args)

if __name__ == '__main__':
    # This guard is crucial for multiprocessing to work correctly on Windows.
    run_experiments()