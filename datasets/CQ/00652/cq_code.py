import cadquery as cq

result = (
cq.Workplane("XY")
.center(0, 0)
.ellipse(3, 5)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')