import cadquery as cq

result = (
cq.Workplane("YZ")
.sketch()
.rect(10, 10)
.push([(5, 5)])
.rect(10, 10, mode="s")
.finalize()
.extrude(1)
)
cq.exporters.export(result, 'GT.stl')