import cadquery as cq

result = (
cq.Workplane("YZ")
.sketch()
.rect(3, 2)
.circle(1)
)
cq.exporters.export(result, 'GT.stl')