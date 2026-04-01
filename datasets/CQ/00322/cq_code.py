import cadquery as cq

result = (
cq.Workplane("XY")
.circle(3)
.workplane(offset=5)
.rect(2, 2)
.loft()
)
cq.exporters.export(result, 'GT.stl')