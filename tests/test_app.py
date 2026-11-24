from streamlit.testing.v1 import AppTest

def test_demo_scores_application():
    app = AppTest.from_file("../app.py", default_timeout=45).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any(item.label == "Default score" for item in app.metric)

    app.multiselect[0].set_value(["mths_since_last_record"])
    app.button[0].click().run()
    assert not app.exception and app.metric

    app.number_input[0].set_value(0.0)
    app.button[0].click().run()
    assert app.error and not app.metric
