import cadquery as cq

result = (
cq.Workplane("YZ")
.ellipse(10, 20)
.extrude(5)
.edges(">X")
.chamfer(2)
.mirror("YZ")
)
cq.exporters.export(result, 'GT.stl')