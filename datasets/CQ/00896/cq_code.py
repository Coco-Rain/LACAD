import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
)
result.exportSvg("box_shape.svg")
cq.exporters.export(result, 'GT.stl')