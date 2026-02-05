SELECT
  d.d_year,
  c.c_city,
  s.s_city,
  SUM(lo.lo_revenue) AS revenue
FROM lineorder lo
JOIN customer c
  ON lo.lo_custkey = c.c_custkey
JOIN supplier s
  ON lo.lo_suppkey = s.s_suppkey
JOIN dates d
  ON lo.lo_orderdate = d.d_datekey
WHERE c.c_city IN (249, 244)
  AND s.s_city IN (249, 244)
  AND lo.lo_orderdate >= 19971201
  AND lo.lo_orderdate <= 19971231
GROUP BY d.d_year, c.c_city, s.s_city;

