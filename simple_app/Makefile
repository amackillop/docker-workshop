URL ?= $(url)

# Generate a random file that is 1 Gb in size.
big-file:
	dd if=/dev/zero of=dataset.txt count=1024 bs=1048576

# Build the docker image
build:
	docker build -f Dockerfile.v2 -t mobilenet .

# Run the app locally
run:
	cd src && flask run

# Run the app in a container
docker-run:
	docker run -it --rm --name mobilenet -p 5000:5000 --gpus "device=0" mobilenet bash -c "cd src && flask run -h 0.0.0.0 -p 5000"

# Given a URL, make a prediction
# make prediction url="<some-url">
prediction:
	curl --header "Content-Type: application/json" \
		 --request POST \
		 --data '{"url": "$(URL)"}' \
		 http://0.0.0.0:5000/predict
