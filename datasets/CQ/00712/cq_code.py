import cadquery as cq

result = (
cq.Workplane("YZ")
.center(1, 1)
.ellipse(10, 2)
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')