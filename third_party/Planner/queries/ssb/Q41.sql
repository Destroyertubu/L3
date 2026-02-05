SELECT
  d.d_year,
  c.c_nation,
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
WHERE c.c_region = 1
  AND s.s_region = 1
  AND (p.p_mfgr = 1 OR p.p_mfgr = 2)
GROUP BY d.d_year, c.c_nation;

