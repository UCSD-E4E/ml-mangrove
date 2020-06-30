# Identify Mangrove Deforestation using Pixel Difference with Daily Satellite Imagery  

Mangrove deforestation often goes unnoticed over long periods of time. Being able to detect change in mangroves will alert authorities to protect mangrove areas. This project compares the pixel difference in mangrove by looking at temporal satellite imagery using Planet Labs. Currently, project only compares pixel difference of entire image, but the goal is to have it compare only mangrove. Longterm, the goal is to have it compare daily imagery rather than weekly.

### Running in Docker


```
docker build -y dockername .
docker run -it dockername
```

When in docker container: 
```
python3 detectcv.py
```


## Libraries/APIs

* OpenCV3
* Imutils
* Scikit-Image
* Planet Labs CLI

## Authors

**Catherine Lee** - *UCSD E4E 2018* 



