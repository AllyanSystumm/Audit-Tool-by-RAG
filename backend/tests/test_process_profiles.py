from backend.generation.process_profiles import detect_process_type, get_profile


def test_detect_process_type_verification():
    t = detect_process_type(filename="Verification.docx", text="Verification Plan and Verification Specification")
    assert t == "verification"


def test_detect_process_type_joint_review():
    t = detect_process_type(filename="Joint review_12_12.docx", text="Joint Review Planning and Joint Review Report")
    assert t == "joint_review"


def test_detect_process_type_configuration_management():
    t = detect_process_type(filename="Configuration+Management.docx", text="Configuration Management Plan, baseline, Change Log")
    assert t == "configuration_management"

def test_detect_verification_not_misclassified_by_cm_mentions():
    text = (
        "Verification Process\n"
        "Verification Plan\n"
        "Verification Test Specification\n"
        "The Verification environment shall be put under Configuration Management.\n"
    )
    t = detect_process_type(filename="Verification.docx", text=text)
    assert t == "verification"


def test_get_profile_auto_none_for_unrelated():
    p = get_profile(process_type="auto", filename="random.docx", text="hello world")
    assert p is None


