IMAGE=odahub/streamlit-spiacs:$(shell git describe --tags --always)

build: 
	docker build . -t $(IMAGE)

push: build
	docker push $(IMAGE)

run: build
	docker run -it -p 8000:8000 $(IMAGE)

deploy:
	helm upgrade --install  streamlit-acs . \
		-f values-unige-dstic-production.yaml \
		--set image.tag="$(shell git describe --tags --always)" 