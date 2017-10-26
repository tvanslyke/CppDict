
BOOST_AUTO_TEST_CASE( test_eq )
{
	std::cerr << "test_eq" << std::endl;
	auto kvps = get_kvps();
	dict_t d1(kvps.begin(), kvps.end());
	dict_t d2(kvps.begin(), kvps.end() - (kvps.size() / 2));	
	BOOST_TEST(d1 != d2);
	BOOST_TEST(not (d1 == d2));

	d2 = d1;
	BOOST_TEST(d1 == d2);
	BOOST_TEST(not (d1 != d2));

	d1.rehash(0);
	BOOST_TEST(d1 == d2);
	BOOST_TEST(not (d1 != d2));

	
	d2 = d1;
	d2.rehash(1024);
	BOOST_TEST(d1 == d2);
	BOOST_TEST(not (d1 != d2));

	
}
