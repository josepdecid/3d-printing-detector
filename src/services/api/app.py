import argparse
import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from flask import Flask, request, Response

from eval import Evaluator


def create_app(checkpoint_path: Optional[str], device: str) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/process_image', methods=['POST'])
    def process_image():
        img_b64_data = request.json.get('img', None)
        if img_b64_data is None:
            return Response('An image to process is required', status=400)

        img_data_decoded = base64.b64decode(img_b64_data)
        img_array = np.array(Image.open(BytesIO(img_data_decoded)))

        if checkpoint_path is not None:
            checkpoints_folder = os.environ.get('CHECKPOINTS_PATH')
            checkpoint_full_path = os.path.join(checkpoints_folder, checkpoint_path)
        else:
            checkpoint_full_path = None

        evaluator = Evaluator(checkpoint_full_path, device)
        class_name = evaluator.evaluate(img_array)

        return Response(class_name, status=200)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='8080', type=str)
    parser.add_argument('--gpu', default=False, type=bool)
    parser.add_argument('--dev', default=False, type=bool,
                        help='Run the app in development (debug) mode. False by default.')
    args = parser.parse_args()

    device = 'gpu' if args.gpu else 'cpu'

    app = create_app(args.checkpoint, device)
    app.run(host=args.host, port=args.port, debug=args.dev)


if __name__ == '__main__':
    main()
