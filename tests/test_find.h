
BOOST_AUTO_TEST_CASE( test_find )
{
	std::cerr << "test_find" << std::endl;
	auto kvps = get_kvps();
	dict_t d1(kvps.begin(), kvps.end() - (kvps.size() / 2));
	auto pos = d1.find(kvps[10].first);
	auto end = d1.end();
	BOOST_TEST((pos != end));
	BOOST_TEST(pos->first == kvps[10].first);
	pos = d1.find(kvps[90].first);
	BOOST_TEST((pos == end));
}
