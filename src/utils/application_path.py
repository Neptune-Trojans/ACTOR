import os.path
import datetime
from pathlib import Path


class ApplicationPath:

    @staticmethod
    def get_application_path(app_name, exp_name):
        home = str(Path.home())
        applications_root_path = os.path.join(*[home, 'Datagen', 'Applications'])
        app_path = os.path.join(applications_root_path, app_name)
        # app_path = os.path.join(app_path, datetime.datetime.now().strftime('%Y_%h_%d_%H_%M'))
        app_path = os.path.join(app_path, exp_name)
        path = Path(app_path)
        path.mkdir(parents=True, exist_ok=True)

        return app_path
