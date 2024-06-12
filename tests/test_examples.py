import os
import subprocess
import unittest


class ExamplesTest(unittest.TestCase):
    def test_examples_run(self):
        here = os.path.dirname(os.path.abspath(__file__))
        examples = f"{here}/../examples"
        scripts = [
            examples + "/" + f for f in os.listdir(examples) if f.endswith(".py")
        ]
        for script in scripts:
            subprocess.check_call(["python", script], stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
