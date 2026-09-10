import unittest
from streamlit.testing.v1 import AppTest


class VolumeUITests(unittest.TestCase):
    def test_examples_edits_clear_and_legacy(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        self.assertFalse(at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,3)
        key='vol_editor_ecoli_out_1'
        at.session_state[key]={'edited_rows':{'0':{'value':4}},'added_rows':[],'deleted_rows':[]}
        at.run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,4)
        at.session_state[key]={'edited_rows':{'0':{'value':5}},'added_rows':[],'deleted_rows':[]}
        at.run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,5)
        at.selectbox[0].set_value('Sporen C. perfringens').run()
        at.selectbox[0].set_value('E. coli').run()
        self.assertEqual(at.session_state['vol_ecoli_out'].iloc[0].value,5)
        at.button[1].click().run()
        self.assertTrue(at.session_state['vol_ecoli_out'].empty)
        at.sidebar.radio[0].set_value('Bisheriger Methodenvergleich').run()
        self.assertFalse(at.exception)

    def test_missing_assumption_blocks_fitting(self):
        at=AppTest.from_file('app.py',default_timeout=30).run()
        at.button[0].click().run()
        at.button[2].click().run()
        self.assertTrue(at.error)
        self.assertNotIn('vol_result_ecoli',at.session_state)


if __name__=='__main__': unittest.main()
