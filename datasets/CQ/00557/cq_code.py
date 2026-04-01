import cadquery as cq

result = (
cq.Workplane("XZ")
.box(20, 5, 10)
.faces(">Y")
.workplane()
.sketch()
.circle(4)
.edges()
.chamfer(0.8)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')