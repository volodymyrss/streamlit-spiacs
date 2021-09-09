NAMESPACE=streamlit-acs
IMAGE=odahub/streamlit-spiacs:$(shell git describe --tags --always)

build: 
	docker build . -t $(IMAGE)

push: build
	docker push $(IMAGE)

run: build
	docker run -it -p 8000:8000 $(IMAGE)

deploy:
	helm upgrade --install  streamlit-acs . \
	        -n $(NAMESPACE)\
		-f values-unige-dstic-production.yaml \
		--set image.tag="$(shell git describe --tags --always)" 

create-secret:
	kubectl create secret generic cdci-secret --from-file=private/cdci-secret.txt -n $(NAMESPACE)
