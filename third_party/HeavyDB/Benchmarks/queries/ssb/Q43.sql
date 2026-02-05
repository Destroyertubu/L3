SELECT
  d.d_year,
  s.s_city,
  p.p_brand1,
  SUM(lo.lo_revenue - lo.lo_supplycost) AS profit
FROM lineorder lo
JOIN customer c
  ON lo.lo_custkey = c.c_custkey
JOIN supplier s
  ON lo.lo_suppkey = s.s_suppkey
JOIN part p
  ON lo.lo_partkey = p.p_partkey
JOIN dates d
  ON lo.lo_orderdate = d.d_datekey
WHERE c.c_nation = 24
  AND s.s_nation = 24
  AND p.p_category = 13
  AND (d.d_year = 1997 OR d.d_year = 1998)
GROUP BY d.d_year, s.s_city, p.p_brand1;

