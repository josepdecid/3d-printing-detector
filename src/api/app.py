import base64
import argparse
from io import BytesIO

import numpy as np
from PIL.Image import Image
from flask import Flask, request, Response


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/process_image', methods=['POST'])
    def process_image():
        image_file = request.files.get('img', None)
        if image_file is None:
            return Response('An image to process is required', status=400)

        img_b64_data = image_file.split(',')[1]
        img_data_decoded = base64.b64decode(img_b64_data)
        img_array = np.array(Image.open(BytesIO(img_data_decoded)))



    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='8080', type=str)
    parser.add_argument('--dev', default=False, type=bool,
                        help='Run the app in development (debug) mode. False by default.')
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.dev_mode)


if __name__ == '__main__':
    main()
