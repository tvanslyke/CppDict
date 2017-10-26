BOOST_AUTO_TEST_CASE( test_count )
{
	std::cerr << "test_count" << std::endl;
	auto kvps = get_kvps(0, 24);
	dict_t d(kvps.begin(), kvps.end() - 12);
	unord_map_t m(kvps.begin(), kvps.end() - 12);

	BOOST_TEST(d.count(kvps[4].first) == 1);
	BOOST_TEST(d.count(kvps[18].first) == 0);
	BOOST_TEST(d.count(kvps[0].first) == 1);
	BOOST_TEST(d.count(kvps[23].first) == 0);


	BOOST_TEST(d.count(kvps[11].first) == m.count(kvps[11].first));
	BOOST_TEST(d.count(kvps[18].first) == m.count(kvps[18].first));
	BOOST_TEST(d.count(kvps[0].first) == m.count(kvps[0].first));
	BOOST_TEST(d.count(kvps[23].first) == m.count(kvps[23].first));
}
