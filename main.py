import avipe_mls

if __name__ == "__main__":
    try:
        config = avipe_mls.config.load("example.yaml")
        avipe_mls.dataset.DatasetDownloader(config.dataset).download()
    except Exception as e:
        print(e)
        exit(-1)
    print(config)