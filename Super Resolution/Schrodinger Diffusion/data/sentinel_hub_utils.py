from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
import os
import json
import tempfile

# Uses Sentinel Hub API to get satellite tif with same bounding box
# Input: Drone Chunk{...}.tif path
# Output: Satellite tif saved as satellite.tif
def call_s2l2a_api(chunk_path):
  URL_REQUEST = "https://services.sentinel-hub.com/api/v1/process"
  START_DATE = "2022-03-01"
  END_DATE = "2022-05-30"

  name = chunk_path.split('/')[-1]
    
  # Get tiff path
  name_list = name.split(' ')
  tiff_name = "Chunk" + name_list[1]
  if len(name_list) > 2:
      tiff_name += "_" + name_list[2]
  tiff_name += ".tif"
  tiff_path = os.path.join(chunk_path, tiff_name)

  # Get Chunk{...}.tif's boundary box
  tiff_raster = rasterio.open(tiff_path)
  tiff_src_crs = tiff_raster.crs
  tiff_src_crs_epsg_code = tiff_src_crs.to_epsg()
  print(f"{chunk_path} epsg_code={tiff_src_crs_epsg_code}")
  tiff_src_bbox = tiff_raster.bounds

  # Find UTM Zone of tiff to use for Sentinel Hub API scaling in meters/pixel
  # Convert bbox from EPSG:4326 to UTM for meter/pixel scaling
  # epsg_code = get_tiff_utm_espg(tiff_path)
  # tiff_dst_crs = f"EPSG:{epsg_code}"
  # tiff_dst_bbox = list(transform_bounds(
  #   tiff_src_crs, tiff_dst_crs, tiff_src_bbox[0], tiff_src_bbox[1], tiff_src_bbox[2], tiff_src_bbox[3], densify_pts=21
  # ))
  # print(f"{tiff_path} has epsg_code={epsg_code}")
  # print(f"Chunk has tiff_src_bbox={tiff_src_bbox} and tiff_dst_bbox={tiff_dst_bbox}")

  # Output satellite tif to Chunk folder
  output_path = os.path.join(chunk_path, 'satellite.tif')
  if os.path.exists(output_path):
    print(f"{name} satellite file already exists\n")
    return
  else:
    print(f"Querying SentinelHub API for {name}")

  # Env file with SentinelHub Client ID and Client Secret
  load_dotenv()
  CLIENT_ID = os.getenv('CLIENT_ID')
  CLIENT_SECRET = os.getenv('CLIENT_SECRET')
  if (CLIENT_ID is None or CLIENT_SECRET is None):
    print("error: CLIENT_ID or CLIENT_SECRET not set")

  client = BackendApplicationClient(client_id=CLIENT_ID)
  oauth = OAuth2Session(client=client)
  # Get an authentication token
  token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
                          client_secret=CLIENT_SECRET, include_client_id=True)

  # Make SentinelHub API call
  evalscript = """
  //VERSION=3
  function setup() {
    return {
      input: [
        "B01","B02","B03","B04","B05","B06",
        "B07","B08","B8A","B09","B11","B12"
      ],
      output: {
        bands: 12,
        sampleType: "AUTO"
      }
    }
  }

  function evaluatePixel(sample) {
    return [
      sample.B01,  2.5*sample.B02,  2.5*sample.B03,  2.5*sample.B04,
      sample.B05,  sample.B06,  sample.B07,  sample.B08,
      sample.B8A,  sample.B09,  sample.B11,  sample.B12
    ];
  }
  """

  json_request = {
    'input': {
      'bounds': {
        'bbox': tiff_src_bbox,
        'properties': {
            'crs': f"http://www.opengis.net/def/crs/EPSG/0/{tiff_src_crs_epsg_code}"
          }
        },
        'data': [
          {
            'type': 'sentinel-2-l2a',
            'dataFilter': {
              'timeRange': {
                'from': f'{START_DATE}T00:00:00Z',
                'to': f'{END_DATE}T23:59:59Z'
              },
              'mosaickingOrder': 'leastCC',
            },
          }
        ]
      },
      'output': {
        'crs': f"http://www.opengis.net/def/crs/EPSG/0/{tiff_src_crs_epsg_code}",
        'resx': 10, #meters per pixel
        'resy': 10, #meters per pixel
        "responses": [
          {
            "identifier": "default",
            "format":     {"type": "image/tiff"}
          }
        ]
      },
  }

  multipart_form = {
    'request': (None, json.dumps(json_request), 'application/json'),
    'evalscript': (None, evalscript, "text/plain"),
  }

  response = oauth.post(
    URL_REQUEST, 
    headers={'Accept': 'image/tiff', 'Authorization': f"Bearer {token['access_token']}"}, 
    files=multipart_form
  )

  if response.status_code == 200:
    with open(output_path, "wb") as f:
      f.write(response.content)
    print(f"✅ TIFF file saved as '{output_path}'\n")
  else:
    print("❌ Request failed with status code:", response.status_code)
    print(response.text)
  
# Get UTM zone from Chunk{...}.tif EPSG:4326 bounds 
def get_tiff_utm_espg(tiff_path):
  with rasterio.open(tiff_path) as src:
    left, bottom, right, top = src.bounds
  lon = (left + right) / 2.0
  lat = (bottom + top) / 2.0

  utm_zone = int((lon + 180) / 6) + 1
  is_north = lat >= 0
  epsg_code = 32600 + utm_zone if is_north else 32700 + utm_zone
  
  return epsg_code

# Reproject a UTM tif to EPSG:4326 CRS. Replaces file in utm_tif_path
def reproject_to_wgs84(utm_tif_path):
  dst_crs = "EPSG:4326"

  with rasterio.open(utm_tif_path) as src:
    transform, width, height = calculate_default_transform(
      src.crs, dst_crs, src.width, src.height, *src.bounds, densify_pts=200
    )
    # update dst metadata
    dst_meta = src.meta.copy()
    dst_meta.update({
      "crs": dst_crs,
      "transform": transform,
      "width": width,
      "height": height,
      "nodata": 0,
    })

    # create temp file and reproject to temp
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
      tmp_path = tmp.name
    with rasterio.open(tmp_path, 'w', **dst_meta) as dst:
      for i in range(1, src.count + 1):
        reproject(
          source=rasterio.band(src, i),
          destination=rasterio.band(dst, i),
          src_transform=src.transform,
          src_crs=src.crs,
          dst_transform=transform,
          dst_crs=dst_crs,
          resampling = Resampling.bilinear,
          src_nodata = 0,
          dst_nodata = 0
        )

    # fill nodata gaps of reprojected temp file for all bands
    with rasterio.open(tmp_path) as dst:
      arr = dst.read()

      # count zeros before fill:
      zeros_before = (arr == 0).sum(axis=(1,2))
      print("Zero-valued pixels (including real data + masked) before:", zeros_before)

      filled = arr.copy()
      for b in dst.indexes:
        # raster bands are 1-indexed, while arr indices are 0-indexed
        band = arr[b-1]
        mask_zero = (band == 0)

        _dist, (inds_row, inds_col) = distance_transform_edt(
          mask_zero,
          return_distances=True,
          return_indices=True
        )

        filled[b-1] = band[inds_row, inds_col]
      
      # count zeros after fill:
      zeros_after = (filled == 0).sum(axis=(1,2))
      print("Zero-valued pixels after fill:", zeros_after)

    # overwrite tmp file's data with filled data
    with rasterio.open(tmp_path, 'w', **dst_meta) as dst:
      dst.write(filled)

    os.replace(tmp_path, utm_tif_path)
