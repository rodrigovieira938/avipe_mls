import avipe_mls

if __name__ == "__main__":
    try:
        config = avipe_mls.config.load("example.yaml")
        dataset = avipe_mls.dataset.create_dataset(config.dataset)
    except Exception as e:
        print(e)
        exit(-1)
    print(config)