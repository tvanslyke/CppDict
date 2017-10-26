
BOOST_AUTO_TEST_CASE( test_equal_range )
{
	std::cerr << "test_equal_range" << std::endl;
	auto kvps = get_kvps();
	dict_t d(kvps.begin(), kvps.end());
	
	{
		auto [a, b] = d.equal_range(kvps[0].first);
		BOOST_TEST(std::distance(a, b) == 1);
		BOOST_TEST(a->first == "zero");
	}
	{
		auto [a, b] = d.equal_range(kvps[127].first);
		BOOST_TEST(std::distance(a, b) == 1);
		BOOST_TEST(a->first == kvps[127].first);
	}
	{
		auto [a, b] = d.equal_range(kvps[33].first);
		BOOST_TEST(std::distance(a, b) == 1);
		BOOST_TEST(a->first == kvps[33].first);
	}
	{
		auto [a, b] = d.equal_range("some nonsense key value");
		BOOST_TEST(std::distance(a, b) == 0);
		d["some nonsense key value"] = -100;
		std::tie(a, b) = d.equal_range("some nonsense key value");
		BOOST_TEST(std::distance(a, b) == 1);
		BOOST_TEST(a->first == "some nonsense key value");
		BOOST_TEST(a->second == -100);
	}

	
}
