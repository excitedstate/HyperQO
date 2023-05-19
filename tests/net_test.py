import sys

sys.path.append("..")


def test_memory_test():
    from src.net import ReplayMemory
    print()
    memory = ReplayMemory(10)

    for i in range(20):
        memory.push(i, i, i, i, i)

    memory.update_weight([0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4, 0.5])
    for i in range(10):
        print(memory.memory[i])

    for m, i in zip(*memory.sample(5)):
        print(m, i)

    # #
