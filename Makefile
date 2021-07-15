IMAGE=odahub/streamlit-spiacs:$(shell git describe --tags --always)

build: 
	docker build . -t $(IMAGE)

push: build
	docker push $(IMAGE)