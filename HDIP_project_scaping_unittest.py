
"""
Student:        Karina Jonina - 10543032
Module:         B8IT110
Module Name:    HDIP PROJECT

Project Objective:           Time Series Forecasting of Cryptocurrency

Task: Unittesting the scrape
"""

import unittest

import HDIP_Project_Scaping

class TestCorona(unittest.TestCase):
    
    def setUp(self):
        self.contents = HDIP_Project_Scaping.get_page_contents()
         
              
    def test_get_page_contents(self):
        self.assertTrue(len(self.contents) > 0)
        
        
    def test_convert_To_soup(self):
        self.assertTrue(HDIP_Project_Scaping.convert_to_soup(self.contents) is not None)
        
#    def test_find_all_links(self):
#        soup =  HDIP_Project_Scaping.convert_to_soup(self.contents)
#        self.assertTrue(len(HDIP_Project_Scaping.convert_to_soup(soup)) > 50)

if __name__ == '__main__':
    unittest.main()    