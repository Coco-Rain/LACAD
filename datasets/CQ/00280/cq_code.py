import cadquery as cq

result = (
cq.Workplane("YZ")
.sketch()
.circle(5)
.circle(3)
.clean()
.finalize()
)
cq.exporters.export(result, 'GT.stl')