import cadquery as cq

result = (
cq.Workplane("XY")
.circle(10)
.workplane(offset=5)
.circle(5)
.loft(combine=True)
)
cq.exporters.export(result, 'GT.stl')