"""CPU-only selection guards; no image builds, pulls or allocation calls."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import ensure_image as image


class ImageSelection(unittest.TestCase):
    def test_defaults_and_explicit_rubin_replacement(self):
        self.assertEqual(image.identity('gb300'), ('2212644', image.BASE))
        self.assertEqual(image.identity('rubin'), ('2212643', image.BASE))
        root = image.BASE.parent / '20260917-campaign-rubin-rerun-j2213753'
        self.assertEqual(image.identity('rubin', '2213753', root), ('2213753', root))
        for platform, job, path in [('rubin', '2213753', image.BASE), ('rubin', '0', root),
                                     ('gb300', '2213753', root), ('rubin', '2213753', '/tmp/campaign')]:
            with self.subTest(platform=platform, job=job, path=path), self.assertRaises(AssertionError):
                image.identity(platform, job, path)

    def test_replacement_plan_has_no_allocation_or_image_access(self):
        root = image.BASE.parent / '20260917-campaign-rubin-rerun-j2213753'
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['ensure_image.py', 'rubin', '--job-id', '2213753', '--campaign-root', str(root)]), \
                patch.object(image, 'allocation', side_effect=AssertionError('remote')), contextlib.redirect_stdout(stream):
            image.main()
        self.assertEqual(json.loads(stream.getvalue())['job'], '2213753')


if __name__ == '__main__':
    unittest.main()
