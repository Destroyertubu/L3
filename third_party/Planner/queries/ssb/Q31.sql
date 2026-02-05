SELECT
  d.d_year,
  c.c_nation,
  s.s_nation,
  SUM(lo.lo_revenue) AS revenue
FROM lineorder lo
JOIN customer c
  ON lo.lo_custkey = c.c_custkey
JOIN supplier s
  ON lo.lo_suppkey = s.s_suppkey
JOIN dates d
  ON lo.lo_orderdate = d.d_datekey
WHERE c.c_region = 2
  AND s.s_region = 2
  AND d.d_year >= 1992
  AND d.d_year <= 1997
GROUP BY d.d_year, c.c_nation, s.s_nation;

