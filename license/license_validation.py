import json
import datetime

class LicenseValidator:
    def __init__(self, license_path):
        self.license_path = license_path

    def validate_license(self):
        try:
            with open(self.license_path, "r") as f:
                license_data = json.load(f)

            # Validate license fields
            expiry_date = datetime.datetime.strptime(license_data["expiry"], "%Y-%m-%d")
            if datetime.datetime.now() > expiry_date:
                raise ValueError("License has expired")

            print(f"License valid for client: {license_data['client']}")
            return True
        except Exception as e:
            print(f"License validation failed: {e}")
            return False

