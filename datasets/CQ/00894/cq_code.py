import cadquery as cq

pts = [(8.0, 0.0), (2.47, 7.6), (-6.47, 4.70), (-6.47, -4.70), (2.47, -7.6)]
result = (
cq.Workplane("XY")
.sketch()
.polygon(pts, angle=15)
.finalize()
.extrude(12)
)
cq.exporters.export(result, 'GT.stl')