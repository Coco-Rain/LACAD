import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
)
result.exportSvg("cylinder_shape.svg")
cq.exporters.export(result, 'GT.stl')