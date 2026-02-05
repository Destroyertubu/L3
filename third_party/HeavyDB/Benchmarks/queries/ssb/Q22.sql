SELECT
  d.d_year,
  p.p_brand1,
  SUM(lo.lo_revenue) AS revenue
FROM lineorder lo
JOIN dates d
  ON lo.lo_orderdate = d.d_datekey
JOIN part p
  ON lo.lo_partkey = p.p_partkey
JOIN supplier s
  ON lo.lo_suppkey = s.s_suppkey
WHERE p.p_brand1 >= 260
  AND p.p_brand1 <= 267
  AND s.s_region = 2
GROUP BY d.d_year, p.p_brand1;

