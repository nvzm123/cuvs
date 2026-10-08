# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from fern.scripts.generate_api_reference import (
    java_signature_at,
    parse_java_members,
)


class GenerateApiReferenceTest(unittest.TestCase):
    def test_java_member_source_line_points_to_declaration(self):
        source = """public class Example {
  /** Plain member. */
  public void plain() {}

  /** Annotated member. */
  @ /* after at-sign */ java /* before dot */ . lang /* before dot */ . Override // The source link should skip this comment.
  /* It should also skip a comment between the annotation and declaration. */
  public String toString() { return "example"; }
}
"""

        members = parse_java_members(source, "Example")

        self.assertEqual(
            [(member.name, member.line) for member in members],
            [("plain", 3), ("toString", 8)],
        )
        self.assertEqual(
            members[1].signature,
            "@ /* after at-sign */ java /* before dot */ . lang "
            "/* before dot */ . Override // The source link should skip this "
            "comment. "
            "/* It should also skip a comment between the annotation and "
            "declaration. */ public String toString()",
        )

    def test_java_source_line_ignores_comments_in_annotation_arguments(self):
        source = """@First(
      value = @Nested(text = "https://example.test/)") /* comment containing ) */,
      other = @Nested( // line comment containing )
          text = "/* not a comment */"))
  public void annotated() {}
"""

        signature, line = java_signature_at(source, 0)

        self.assertEqual(line, 5)
        self.assertEqual(
            signature,
            '@First( value = @Nested(text = "https://example.test/)") '
            "/* comment containing ) */, other = @Nested( // line comment "
            'containing ) text = "/* not a comment */")) '
            "public void annotated()",
        )


if __name__ == "__main__":
    unittest.main()
